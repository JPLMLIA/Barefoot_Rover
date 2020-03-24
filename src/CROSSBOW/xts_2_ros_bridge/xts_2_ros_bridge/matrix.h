// matrix.h - CMatrix template class header
//
// Copyright (C) 2006 Xiroku Inc.
//
// Copied from Xiroku sample application xtsview. See Xiroku distribution
// CD for context.
//
#ifndef _MATRIX_H_
#define _MATRIX_H_

/////////////////////////////////////////////////////////////////////////////
// CMatrix
template <class T> class CMatrix
{
public:
	CMatrix(void) { m_data = NULL; m_nrow = 0; m_ncol = 0; }
	~CMatrix(void) { Deallocate(); }
	BOOL Allocate(int nrow, int ncol);
	void Deallocate(void);
	void Clear(void);
	T* operator [] (int i) { return m_data[i]; }
	BOOL IsEmpty(void) const { return (m_data == NULL); }
	int RowCount(void) const { return m_nrow; }
	int ColumnCount(void) const { return m_ncol; }
private:
	T** m_data;
	int m_nrow;
	int m_ncol;
};

template <class T> 
BOOL CMatrix<T>::Allocate(int nrow, int ncol)
{
	if (nrow <= 0 || ncol <= 0) {
		return FALSE;
	}
	if (m_data) {
		Deallocate();
	}
	size_t size = nrow * ncol * sizeof(T) + nrow * sizeof(T*);
	BYTE *pt = new BYTE [size];
	ZeroMemory(pt, size);
	m_data = (T**) pt;
	pt += nrow * sizeof(T*);
	for (int i = 0; i < nrow; i++) {
		m_data[i] = (T*) pt;
		pt += ncol * sizeof(T);
	}
	m_nrow = nrow;
	m_ncol = ncol;
	return TRUE;
}

template <class T> 
void CMatrix<T>::Deallocate(void)
{
	if (m_data) {
		delete [] (BYTE*) m_data;
		m_data = NULL;
	}
}

template <class T> 
void CMatrix<T>::Clear(void)
{
	if (m_data) {
		size_t size = m_nrow * m_ncol * sizeof(T);
		ZeroMemory(m_data[0], size);
	}
}

#endif //_MATRIX_H_
